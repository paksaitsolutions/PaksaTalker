from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading

def run_server():
    server_address = ('', 8001)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Starting HTTP server on port 8001...")
    httpd.serve_forever()

if __name__ == "__main__":
    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open browser
    webbrowser.open('http://localhost:8001')
    
    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down server...")
