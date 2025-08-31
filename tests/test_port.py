import socket
import sys

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

if __name__ == "__main__":
    port = 5000
    if check_port(port):
        print(f"Port {port} is available")
        
        # Try to create a simple HTTP server
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        
        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Test server is working!')
        
        print(f"Starting test server on port {port}...")
        httpd = HTTPServer(('', port), Handler)
        print(f"Server started on port {port}")
        print("Try accessing http://localhost:5000 in your browser")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()
    else:
        print(f"Port {port} is already in use")
        sys.exit(1)
