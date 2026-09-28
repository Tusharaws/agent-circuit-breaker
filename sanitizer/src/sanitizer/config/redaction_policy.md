# Redaction Policy

Maps a data type (as detected by a scrubber) to one of three actions:
`mask` (replace with a placeholder), `hash` (replace with a stable hash),
or `drop` (remove entirely). Edit this file directly to change scrub
behavior for any data type -- no code change required.

An operator can also override this entire file at runtime without a
redeploy by setting `SANITIZER_REDACTION_POLICY_PATH` to point at an
external file instead.

- email: mask
- phone: mask
- ssn: drop
- credit_card: mask
- api_key: drop
- person: mask
- address: mask
- medical_condition: drop
