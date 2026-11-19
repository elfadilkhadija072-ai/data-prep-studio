"""Point d'entrée :  python run.py   →   http://localhost:8501  (API + interface React compilée)."""
import sys, uvicorn
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    print(f"\n  DataPrep Studio  →  http://localhost:{port}\n")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
