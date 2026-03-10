import mimetypes
import uvicorn
import sys
import os

# Fix MIME types BEFORE anything else — Windows Python defaults .js to text/plain
# which causes browsers to reject module scripts
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
