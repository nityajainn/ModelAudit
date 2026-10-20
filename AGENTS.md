# ModelAudit development rules

- Work in this independent repository only. Do not modify Project 2 or the
  older ModelAudit checkout, their files, credentials, or Git configuration.
- Implement phases in docs/PHASE_PLAN.md order. Later phases stay planned
  until their actual implementation and verification.
- Ask the user for explicit permission before EVERY phase push. Approval for
  one phase never authorizes another. Show the reviewed commit, checks,
  account, destination, branches, and tags before requesting approval.
- Attribute new user-requested commits to the verified nityajainn account.
  Do not add collaborators or co-author trailers without user instruction.
  Do not import old commits, misattribute reused code, or remove required
  third-party notices. GitHub contributor display is derived from history.
- Use real commit timestamps; never backdate or fabricate activity. Record
  draft work and completion dates accurately in the build log.
- Never force-push. Stop if the destination contains conflicting history.
- Use only the secondary remote for approved publication to
  https://github.com/nityajainn/ModelAudit.git.
- Keep datasets, models, secrets, environments, and generated reports out of Git.
