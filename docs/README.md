
## Usage

```bash
cd /path/to/icefall/docs
pip install -r requirements.txt
make clean
make html
cd build/html
python3 -m http.server 8000
```

It prints:

```
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

Open your browser and go to <http://0.0.0.0:8000/> to view the generated
documentation.

Done!

**Hint**: You can change the port number when starting the server.
