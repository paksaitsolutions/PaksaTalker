from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify({"status": "success", "message": "Flask server is running!"})

if __name__ == '__main__':
    port = 5001  # Changed from 5000 to 5001
    print(f"Starting Flask server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=port, debug=True)
