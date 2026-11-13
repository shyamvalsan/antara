# Antara backend

Cloudflare Worker `antara-guestbook` + D1 database `antara`: hearts and a moderated guest book.
Live at https://antara-guestbook.shiny-mountain-4a1e.workers.dev

## Deploy
    cd backend
    npx wrangler d1 create antara            # once; put the id in wrangler.toml
    npx wrangler d1 execute antara --remote --file=schema.sql
    npx wrangler secret put ADMIN_TOKEN      # openssl rand -hex 32
    npx wrangler secret put IP_SALT          # openssl rand -hex 32
    npx wrangler deploy
The token and URL are in `.admin-token` (git-ignored).

## Allow-list of works (re-seed after the roster changes)
    curl -X PUT $URL/admin/works -H "Authorization: Bearer $ADMIN_TOKEN" -d @works.json
`works.json` holds every exhibition slug and work id (file stem, `_` to `-`) from `curator/programme.toml`.

## Moderate
Open `$URL/admin`, paste the token (kept in sessionStorage only). Pending entries have Approve / Reject; Approved and Rejected tabs allow switching and Delete. Nothing is public until approved.

## API (JSON; CORS for github.io and localhost:8765 / 127.0.0.1:8765)
- `GET /hearts?visitor=<uuid>` -> `{counts:{slug:n}, mine:[slug]}` (`mine` empty without `visitor`)
- `POST /hearts/<slug>` `{visitor, on:true|false}` -> `{count, on}`. 404 unknown work, 400 bad slug/visitor, 429 over 60 writes/IP/hour or more than 3 visitor ids per slug per IP per day.
- `POST /guestbook` `{name<=60, message 2-1000, from?<=60, work?, visitor, website:""}` -> 201 `{ok:true,pending:true}`. 400 invalid/honeypot/over 2 links, 413 body over 4 KB, 429 over 3/hour or 10/day per IP.
- `GET /guestbook?limit=50&before=<id>` -> `{entries:[{id,name,from,message,work,created}], next}` (approved, newest first, cached 30 s; `created` is unix seconds)
- `POST /visit` `{visitor}` -> `{visitors, new}` (distinct browser ids; upsert; 400 bad id, 429 over 30 new ids per IP per day)
- `GET /stats` -> `{hearts, entries, visitors}`
- Admin (`Authorization: Bearer <token>`): `GET /admin/entries?status=pending|approved|rejected`, `POST /admin/entries/<id>` `{status:"approved"|"rejected"}`, `DELETE /admin/entries/<id>`, `PUT /admin/works` `{slugs:[...]}`; `GET /admin` is the moderation page.

IPs are never stored; only SHA-256(IP_SALT|ip).
