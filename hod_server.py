from flask import Flask, render_template_string, request, redirect, url_for
import os
import datetime
from email_sender import send_email
from timetable_manager import get_parent_email

app = Flask(__name__)

# In-memory storage for violations (use a DB in production)
violations = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HOD Dashboard - Student Violations</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .card { border: 1px solid #ccc; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
        .btn { padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 3px; }
        .btn-danger { background-color: #dc3545; }
        img { max-width: 200px; display: block; margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>HOD Approval Dashboard</h1>
    {% if not violations %}
        <p>No pending violations.</p>
    {% else %}
        {% for v in violations %}
            <div class="card">
                <h3>{{ v.name }} ({{ v.roll }})</h3>
                <p><strong>Class Missed:</strong> {{ v.subject }}</p>
                <p><strong>Time:</strong> {{ v.time }}</p>
                <img src="/static/captures/{{ v.image_file }}" alt="Student Photo">
                <form action="/approve/{{ loop.index0 }}" method="post" style="display:inline;">
                    <button type="submit" class="btn">Approve & Notify Parents</button>
                </form>
                <form action="/reject/{{ loop.index0 }}" method="post" style="display:inline;">
                    <button type="submit" class="btn btn-danger">Reject / Ignore</button>
                </form>
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, violations=violations)

@app.route('/report', methods=['POST'])
def report_violation():
    data = request.json
    # data: {roll, name, subject, image_file}
    data['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violations.append(data)
    return {"status": "received"}, 200

@app.route('/approve/<int:vid>', methods=['POST'])
def approve(vid):
    if 0 <= vid < len(violations):
        v = violations.pop(vid)
        parent_email = get_parent_email(v['roll'])
        
        if parent_email:
            subject = f"Attendance Alert: {v['name']} missed {v['subject']}"
            body = f"""
            Dear Parent,
            
            Your ward {v['name']} (Roll No: {v['roll']}) was found outside class during {v['subject']} at {v['time']}.
            
            Please see the attached photo.
            
            Regards,
            College Administration
            """
            image_path = os.path.join('static', 'captures', v['image_file'])
            send_email(subject, body, to_email=parent_email, image_path=image_path)
            print(f"Notification sent to {parent_email}")
        else:
            print(f"No parent email found for {v['roll']}")
            
    return redirect(url_for('index'))

@app.route('/reject/<int:vid>', methods=['POST'])
def reject(vid):
    if 0 <= vid < len(violations):
        violations.pop(vid)
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure static/captures exists
    os.makedirs('static/captures', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
