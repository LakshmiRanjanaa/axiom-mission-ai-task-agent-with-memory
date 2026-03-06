from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

class TaskAgent:
    def __init__(self, db_path='tasks.db'):
        self.db_path = db_path
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with tasks table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                estimated_duration INTEGER NOT NULL,
                actual_duration INTEGER,
                priority TEXT DEFAULT 'medium',
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        # Add sample data if table is empty
        cursor.execute('SELECT COUNT(*) FROM tasks')
        if cursor.fetchone()[0] == 0:
            sample_tasks = [
                ('Review resume', 'admin', 15, 12, 'high', True),
                ('Phone screening', 'interview', 30, 35, 'high', True),
                ('Write job posting', 'content', 45, 60, 'medium', True),
                ('Research candidates', 'research', 60, 45, 'medium', True)
            ]
            
            cursor.executemany('''
                INSERT INTO tasks (title, category, estimated_duration, actual_duration, priority, completed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', sample_tasks)
        
        conn.commit()
        conn.close()
    
    def train_model(self):
        """Train ML model on completed tasks"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT category, estimated_duration, priority, actual_duration FROM tasks WHERE completed = TRUE',
            conn
        )
        conn.close()
        
        if len(df) < 3:  # Need minimum data for training
            return False
        
        # Prepare features
        categories = self.label_encoder.fit_transform(df['category'])
        priorities = LabelEncoder().fit_transform(df['priority'])
        
        X = np.column_stack([
            categories,
            df['estimated_duration'],
            priorities
        ])
        y = df['actual_duration']
        
        self.model.fit(X, y)
        self.is_trained = True
        return True
    
    def predict_duration(self, category, estimated_duration, priority):
        """Predict actual duration based on task features"""
        if not self.is_trained:
            if not self.train_model():
                return estimated_duration  # Fallback to estimate
        
        try:
            # Encode category (handle unseen categories)
            try:
                cat_encoded = self.label_encoder.transform([category])[0]
            except ValueError:
                cat_encoded = 0  # Default for unseen category
            
            priority_map = {'low': 0, 'medium': 1, 'high': 2}
            priority_encoded = priority_map.get(priority, 1)
            
            prediction = self.model.predict([[cat_encoded, estimated_duration, priority_encoded]])
            return max(5, int(prediction[0]))  # Minimum 5 minutes
        except:
            return estimated_duration

# Initialize agent
agent = TaskAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tasks', methods=['GET', 'POST'])
def handle_tasks():
    if request.method == 'GET':
        conn = sqlite3.connect(agent.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
        tasks = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': task[0], 'title': task[1], 'category': task[2],
            'estimated_duration': task[3], 'actual_duration': task[4],
            'priority': task[5], 'completed': bool(task[6])
        } for task in tasks])
    
    elif request.method == 'POST':
        data = request.json
        
        # Get AI prediction
        predicted_duration = agent.predict_duration(
            data['category'], data['estimated_duration'], data['priority']
        )
        
        conn = sqlite3.connect(agent.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tasks (title, category, estimated_duration, priority)
            VALUES (?, ?, ?, ?)
        ''', (data['title'], data['category'], data['estimated_duration'], data['priority']))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'predicted_duration': predicted_duration,
            'message': f'AI suggests this task might take {predicted_duration} minutes (vs your estimate of {data["estimated_duration"]} minutes)'
        })

@app.route('/api/tasks/<int:task_id>/complete', methods=['POST'])
def complete_task(task_id):
    data = request.json
    actual_duration = data['actual_duration']
    
    conn = sqlite3.connect(agent.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE tasks 
        SET completed = TRUE, actual_duration = ?, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (actual_duration, task_id))
    conn.commit()
    conn.close()
    
    # Retrain model with new data
    agent.train_model()
    
    return jsonify({'success': True, 'message': 'Task completed! AI model updated.'})

if __name__ == '__main__':
    app.run(debug=True)