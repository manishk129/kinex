# Kinex

Kinex is a voice-first family assistant that helps families stay organized, connected, and on time. It can talk naturally with each family member, remember useful context, and help with everyday coordination like plans, reminders, tasks, and shared updates.

## What Kinex does

- **Natural voice conversations** - talk to Kinex like a helpful family assistant.
- **Family-aware memory** - remembers useful preferences, commitments, and shared context across sessions.
- **Private by design** - keeps personal memories scoped to the right person, with shared family context when appropriate.
- **Everyday coordination** - helps with schedules, reminders, tasks, and family updates.
- **Calendar-ready** - can be connected to Google Calendar tools for scheduling and availability.

## Run locally

```bash
bun run setup
bun run dev
```

Open [http://localhost:3000](http://localhost:3000) and start a conversation.

## Project layout

- `web/` - Kinex web experience
- `server/` - Kinex voice agent backend
- `server/src/agent.py` - assistant prompt, memory, and tool wiring

## License

Released under the [MIT License](./LICENSE).
