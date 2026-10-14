"""Read-only cloud readiness checks; credentials never enter tracker payloads."""
import base64
import json
import netrc
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from . import PROJECT
from .io import TrackingError

API = 'https://api.wandb.ai'


def proxy_environment(proxy, environment=None):
    """Scope proxy routing to this tracker and its transport children."""
    env = dict(os.environ if environment is None else environment)
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path not in ('', '/'):
            raise TrackingError('Tracker proxy must be an HTTP/HTTPS proxy URL')
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            env[name] = proxy
        bypass = set((env.get('NO_PROXY') or env.get('no_proxy') or '').split(','))
        bypass.update(('localhost', '127.0.0.1', '::1'))
        env['NO_PROXY'] = env['no_proxy'] = ','.join(sorted(bypass - {''}))
    return env


def credential():
    value = os.environ.get('WANDB_API_KEY')
    if value:
        return value
    try:
        auth = netrc.netrc().authenticators('api.wandb.ai')
        return auth[2] if auth else None
    except (OSError, netrc.NetrcParseError):
        return None


def readiness(entity):
    if not entity:
        raise TrackingError('Set WANDB_ENTITY or --entity to your personal/team slug')
    if os.environ.get('WANDB_BASE_URL', API).rstrip('/') != API:
        raise TrackingError('This tracker targets W&B Cloud; custom API hosts require explicit adaptation')
    key = credential()
    if not key:
        raise TrackingError('Run wandb login in your terminal; never send API keys in chat')
    payload = {'query': '''query FGTrackerReadiness($entity: String!, $project: String!) {
      viewer { username }
      project(name: $project, entityName: $entity) { name access }
    }''', 'variables': {'entity': entity, 'project': PROJECT}}
    request = urllib.request.Request(API+'/graphql', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json',
            'Authorization': 'Basic '+base64.b64encode(('api:'+key).encode()).decode()})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read(1024 * 1024))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TrackingError('W&B credential/project readiness request failed; local spool remains intact') from exc
    if result.get('errors'):
        raise TrackingError('W&B readiness query rejected; verify login, Entity and existing private Project')
    data = result.get('data') or {}
    if not data.get('viewer') or not data.get('project'):
        raise TrackingError('W&B login or Project is unavailable; create/confirm the private Project in the web UI')
    visibility = str(data['project'].get('access', '')).upper()
    if visibility != 'PRIVATE':
        raise TrackingError('Project is not verified PRIVATE; no experiment data will be uploaded')
    return dict(status='PRIVATE_PROJECT_AUTHENTICATED', entity=entity, project=PROJECT,
        visibility=visibility, credential_valid=True,
        project_url=f'https://wandb.ai/{entity}/{PROJECT}')
