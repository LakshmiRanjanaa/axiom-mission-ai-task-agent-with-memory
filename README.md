# AI Task Agent with Memory

An intelligent task management agent that learns from user behavior and suggests optimal task scheduling.

## Features
- SQLite database for persistent storage
- Machine learning for task duration prediction
- Web interface for task management
- Agent learns from completed task patterns

## Setup
1. Install dependencies: `pip install flask scikit-learn pandas numpy`
2. Run the application: `python app.py`
3. Open browser to `http://localhost:5000`

## How it works
1. Add tasks with estimated durations
2. Mark tasks as completed with actual time taken
3. The AI agent learns from your patterns and suggests better time estimates
4. Get personalized scheduling recommendations

## Next Steps
- Implement priority-based scheduling
- Add more sophisticated ML models
- Include calendar integration
- Add user authentication