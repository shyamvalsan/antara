CREATE TABLE IF NOT EXISTS works (slug TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS hearts (
  slug TEXT NOT NULL, visitor TEXT NOT NULL, created INTEGER NOT NULL,
  PRIMARY KEY (slug, visitor)
);
CREATE INDEX IF NOT EXISTS hearts_slug ON hearts(slug);
-- heart write log, for rate limits only (salted IP hash, never raw IP)
CREATE TABLE IF NOT EXISTS heart_events (
  iphash TEXT NOT NULL, slug TEXT NOT NULL, visitor TEXT NOT NULL, ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS heart_events_ip ON heart_events(iphash, ts);
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL, place TEXT, message TEXT NOT NULL, work TEXT,
  visitor TEXT, iphash TEXT,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  created INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_status ON entries(status, id);
CREATE INDEX IF NOT EXISTS entries_ip ON entries(iphash, created);
-- visitors: one row per browser id (first and last visit), plus a log of new ids per salted IP hash for rate limiting
CREATE TABLE IF NOT EXISTS visits (visitor TEXT PRIMARY KEY, first INTEGER NOT NULL, last INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS visit_events (iphash TEXT NOT NULL, ts INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS visit_events_ip ON visit_events(iphash, ts);
