"""Script de lancement unique du serveur de production DataPrep Studio."""
import os
import sys
import subprocess
import webbrowser
import uvicorn

# Forcer la sortie console en UTF-8 sur Windows pour supporter les caractères spéciaux
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
TOOLS_NODE_DIR = os.path.join(ROOT_DIR, ".tools", "nodejs")


def find_node_npm():
    """Détecte node et npm dans .tools ou dans le PATH système."""
    node_exe = os.path.join(TOOLS_NODE_DIR, "node.exe")
    npm_cmd = os.path.join(TOOLS_NODE_DIR, "npm.cmd")
    if os.path.exists(node_exe) and os.path.exists(npm_cmd):
        # Ajouter le dossier nodejs au PATH de l'environnement courant
        if TOOLS_NODE_DIR not in os.environ.get("PATH", ""):
            os.environ["PATH"] = TOOLS_NODE_DIR + os.pathsep + os.environ.get("PATH", "")
        return node_exe, npm_cmd
    return "node", "npm"


def ensure_frontend_built():
    """Vérifie si le frontend React est compilé dans dist/, sinon le compile automatiquement."""
    index_html = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_html):
        print(f"[*] Frontend compilé détecté dans {DIST_DIR}")
        return

    print("[*] Le frontend n'est pas encore compilé. Lancement automatique du build...")
    node_exe, npm_cmd = find_node_npm()
    
    # npm install si node_modules absent
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print("[*] Installation des dépendances frontend (npm install)...")
        subprocess.run([npm_cmd, "install"], cwd=FRONTEND_DIR, check=True)

    print("[*] Compilation des assets React avec Vite (npm run build)...")
    subprocess.run([npm_cmd, "run", "build"], cwd=FRONTEND_DIR, check=True)
    print("[*] Build terminé avec succès.")


def main():
    print("=" * 65)
    print("        🚀 DÉMARRAGE DE DATAPREP STUDIO (Production)")
    print("=" * 65)
    
    ensure_frontend_built()

    port = 8501
    # Vérification de disponibilité du port
    import socket
    for candidate in [8501, 8502, 8503, 8000]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", candidate))
            s.close()
            port = candidate
            break
        except Exception:
            continue

    url = f"http://localhost:{port}"
    print(f"\n[+] Serveur disponible sur : {url}")
    print(f"[+] API Swagger documentée sur : {url}/docs")
    print("[+] Arrêt du serveur : Ctrl+C dans ce terminal.\n")

    # Ouvrir le navigateur automatiquement
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Lancer le serveur ASGI Uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
