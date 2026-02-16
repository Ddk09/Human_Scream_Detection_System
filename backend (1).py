from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("scream_event")
def forward_event(data):
    print("FORWARDING EVENT:", data)  
    socketio.emit("scream_event", data)  


if __name__ == "__main__":
    print("Backend running on http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000)







