import http.server
import json
import sys
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]
REDIRECT_PORT = 4100
REDIRECT_PATH = "/code"


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_google_refresh_token.py <path_to_credentials.json>")
        sys.exit(1)

    creds_path = Path(sys.argv[1])
    creds = json.loads(creds_path.read_text())
    c = creds.get("web") or creds.get("installed") or creds
    client_id = c["client_id"]
    client_secret = c["client_secret"]

    auth_params = {
        "client_id": client_id,
        "redirect_uri": f"http://localhost:{REDIRECT_PORT}{REDIRECT_PATH}",
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(auth_params)
    print("Open this URL in a browser (on your laptop):")
    print()
    print(auth_url)
    print()
    print(f"After consent, Google will redirect to http://localhost:{REDIRECT_PORT}{REDIRECT_PATH}?code=...")
    print("Paste the FULL redirected URL (or just the ?code= value) here:")
    url_or_code = input("> ").strip()

    if url_or_code.startswith("http"):
        parsed = urllib.parse.urlparse(url_or_code)
        code = urllib.parse.parse_qs(parsed.query).get("code", [""])[0]
    else:
        code = url_or_code

    token_req = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": f"http://localhost:{REDIRECT_PORT}{REDIRECT_PATH}",
        "grant_type": "authorization_code",
    }).encode()

    resp = urllib.request.urlopen(
        "https://oauth2.googleapis.com/token", data=token_req
    )
    tok = json.loads(resp.read())

    print()
    print("Add these to your .env:")
    print(f"GOOGLE_CLIENT_ID={client_id}")
    print(f"GOOGLE_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={tok['refresh_token']}")


if __name__ == "__main__":
    main()
