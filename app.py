import os
import json
import re
import sqlite3
import threading
import traceback
from flask import Flask, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from spotapi import Podcast

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='public')

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# SQLite setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'saved_queries.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS saved_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            result_count INTEGER,
            episodes TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Initialize OpenRouter client (uses OpenAI SDK)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
)


# Pydantic models for structured output (LLM returns just IDs)
class RelevantEpisodeIds(BaseModel):
    episode_ids: List[int]


def extract_show_id(input_str):
    """Extract Spotify show ID from URL or return raw ID"""
    url_match = re.search(r'open\.spotify\.com/show/([a-zA-Z0-9]+)', input_str)
    if url_match:
        return url_match.group(1)
    # Assume it's already an ID
    if re.match(r'^[a-zA-Z0-9]+$', input_str):
        return input_str
    raise ValueError(f'Invalid podcast URL or ID: {input_str}')


def send_sse_event(event_type, data):
    """Format data as SSE event"""
    return f'data: {json.dumps({"type": event_type, "data": data})}\n\n'


def fetch_all_episodes(show_id):
    """Fetch ALL episodes from a podcast using SpotAPI (paginated)"""
    podcast = Podcast(podcast=show_id)
    all_episodes = []

    for batch in podcast.paginate_podcast():
        for episode in batch:
            all_episodes.append(episode)

    return all_episodes


def extract_episode_data(episode):
    """Extract structured data from SpotAPI nested format"""
    entity = episode.get('entity', {})
    uri = entity.get('_uri', '')
    episode_id = uri.split(':')[-1] if ':' in uri else ''
    
    data = entity.get('data', {})
    title = data.get('name', 'Unknown Title')
    
    description = data.get('description', '')
    if not description:
        description = data.get('subtitle', '') or data.get('abstract', '') or ''
    
    show_name = episode.get('show_name', 'Unknown')
    url = f'https://open.spotify.com/episode/{episode_id}' if episode_id else 'N/A'
    
    return {
        'title': title,
        'url': url,
        'show_name': show_name,
        'description': description
    }


def format_episode_for_prompt(episode, index):
    """Format an episode for the LLM prompt - just title and brief description"""
    entity = episode.get('entity', {})
    data = entity.get('data', {})
    title = data.get('name', 'Unknown Title')
    
    description = data.get('description', '')
    if not description:
        description = data.get('subtitle', '') or data.get('abstract') or ''
    description = description[:config['llm']['description_max_chars']]
    
    return f'{title} - {description}'


def find_relevant_episodes(episodes, topic, result_count=None):
    """Use OpenRouter LLM to find relevant episodes by ID"""
    llm_config = config['llm']
    prompts_config = config['prompts']
    
    # Use provided result_count or fall back to config
    if result_count is None:
        result_count = llm_config['results_count']
    
    # Debug: show first few episodes
    print(f'\n[DEBUG] Total episodes: {len(episodes)}')
    
    # Prepare episode list for the prompt (limit to avoid token limits)
    max_episodes = llm_config['max_episodes_in_prompt']
    limited_episodes = episodes[:max_episodes]
    
    # Show first 5 for debugging
    formatted = [format_episode_for_prompt(ep, i) for i, ep in enumerate(limited_episodes[:5])]
    print(f'[DEBUG] First 5 formatted episodes:')
    for i, f in enumerate(formatted):
        print(f'  {i+1}. {f}')
    
    # Build numbered list for prompt
    episode_list = '\n'.join([
        f'{i + 1}. {format_episode_for_prompt(ep, i)}'
        for i, ep in enumerate(limited_episodes)
    ])
    print(f'[DEBUG] Prompt size: {len(episode_list)} characters')

    # Build prompt from template
    prompt = prompts_config['user_template'].format(
        topic=topic,
        episode_list=episode_list,
        results_count=result_count
    )

    # Try models in order
    models = llm_config['models']
    last_error = None

    for model in models:
        try:
            completion = openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': prompts_config['system']
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=llm_config['temperature'],
                reasoning_effort=llm_config["reasoning_effort"],
            )

            raw_content = completion.choices[0].message.content
            print(f'\n[DEBUG] Raw LLM response:\n{raw_content}\n')
            
            result = json.loads(raw_content)
            
            # Validate with Pydantic
            validated = RelevantEpisodeIds(**result)
            print(f'[DEBUG] Validated episode IDs: {validated.episode_ids}')
            
            # Map IDs back to full episode data
            relevant_episodes = []
            for idx in validated.episode_ids:
                # Convert 1-based index to 0-based
                array_idx = idx - 1
                if 0 <= array_idx < len(limited_episodes):
                    ep_data = extract_episode_data(limited_episodes[array_idx])
                    relevant_episodes.append(ep_data)
            
            print(f'[DEBUG] Mapped to {len(relevant_episodes)} episodes')
            return relevant_episodes, model

        except Exception as e:
            last_error = e
            if model == models[-1]:
                raise Exception(f'All models failed. Last error:\n{str(last_error)}\n{traceback.format_exc()}')


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('public', 'index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """Search for relevant podcast episodes using SSE for progress updates"""
    data = request.get_json()
    topic = data.get('topic', '').strip()
    result_count = data.get('result_count', config['llm']['results_count'])

    if not topic:
        return jsonify({'error': 'Topic is required'}), 400

    def generate():
        try:
            # Step 1: Extract podcast IDs
            yield send_sse_event('status', '⏳ Step 1/3: Extracting podcast IDs from config...')
            show_ids = []
            for p in config['podcasts']:
                try:
                    show_ids.append(extract_show_id(p))
                except ValueError as e:
                    yield send_sse_event('error', {
                        'message': str(e),
                        'stack': traceback.format_exc()
                    })
                    return
            yield send_sse_event('status', f'✅ Step 1/3: Found {len(show_ids)} podcasts')

            # Step 2: Fetch all episodes
            yield send_sse_event('status', '⏳ Step 2/3: Fetching episodes from Spotify...')
            all_episodes = []

            for show_id in show_ids:
                try:
                    yield send_sse_event('status', f'   → Fetching podcast: {show_id}')
                    episodes = fetch_all_episodes(show_id)

                    # Add show name to each episode
                    for ep in episodes:
                        ep['show_name'] = ep.get('show', {}).get('name', show_id)

                    all_episodes.extend(episodes)
                    yield send_sse_event('status', f'   → Got {len(episodes)} episodes')
                except Exception as e:
                    yield send_sse_event('error', {
                        'message': f'Failed to fetch podcast {show_id}: {str(e)}',
                        'stack': traceback.format_exc()
                    })
                    return

            yield send_sse_event('status', f'✅ Step 2/3: Fetched {len(all_episodes)} episodes total')

            # Step 3: Find relevant episodes using OpenRouter
            yield send_sse_event('status', '⏳ Step 3/3: Sending to OpenRouter LLM with structured output...')

            # Run LLM call in a thread so we can send keepalive pings
            # (Render proxy kills connections after ~30s of no SSE data)
            llm_result = {}
            def llm_worker():
                try:
                    llm_result['episodes'], llm_result['model'] = find_relevant_episodes(all_episodes, topic, result_count)
                except Exception as e:
                    llm_result['error'] = e

            thread = threading.Thread(target=llm_worker)
            thread.start()

            while thread.is_alive():
                thread.join(timeout=15)
                if thread.is_alive():
                    yield send_sse_event('status', '⏳ Still processing... (LLM is analyzing episodes)')

            if 'error' in llm_result:
                raise llm_result['error']

            relevant_episodes = llm_result['episodes']
            model_used = llm_result['model']
            yield send_sse_event('status', f'✅ Step 3/3: Done! Used model: {model_used}')

            # Send results
            yield send_sse_event('result', {
                'topic': topic,
                'episodes': relevant_episodes
            })

            # Save to database
            conn = None
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    'INSERT INTO saved_queries (topic, result_count, episodes) VALUES (?, ?, ?)',
                    (topic, result_count, json.dumps(relevant_episodes))
                )
                conn.commit()
            except Exception as db_err:
                print(f'[WARN] Failed to save query: {db_err}')
            finally:
                if conn:
                    conn.close()

        except Exception as e:
            yield send_sse_event('error', {
                'message': str(e),
                'stack': traceback.format_exc()
            })

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/config')
def get_config():
    """Get current podcast configuration"""
    return jsonify(config)


@app.route('/api/saved-queries')
def get_saved_queries():
    """List all saved queries (most recent first)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT id, topic, result_count, created_at FROM saved_queries ORDER BY id DESC'
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route('/api/saved-queries/<int:query_id>')
def get_saved_query(query_id):
    """Get a single saved query with its full results"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        'SELECT id, topic, result_count, episodes, created_at FROM saved_queries WHERE id = ?',
        (query_id,)
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Query not found'}), 404

    result = dict(row)
    result['episodes'] = json.loads(result['episodes'])
    return jsonify(result)


if __name__ == '__main__':
    print('\n🎧 Podcast Fetcher running at http://localhost:3000')
    print('\n📋 Configured podcasts:')
    for i, p in enumerate(config['podcasts'], 1):
        print(f'   {i}. {p}')
    print('\n🔑 Make sure your .env file has:')
    print('   - OPENROUTER_API_KEY\n')

    app.run(host='0.0.0.0', port=3000, debug=True)
