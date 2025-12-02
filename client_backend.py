from flask import Flask,request, jsonify
import requests
import os

app=Flask(__name__)

@app.route('/api/send-local-model', methods=['POST'])
def send_local_model():
    file = request.files.get("file")
    client_id = request.form.get("client_id")
    federated_server_url = request.form.get("federated_server_url")
    cur_round=request.form.get("cur_round")
    dataset_size=request.form.get("dataset_size")

    if file is None:
        return jsonify({"success": False, "error": "No file received"}), 400

    upload_url = f"{federated_server_url}/api/upload-client-weights"

    print(f"[CLIENT] Forwarding model to {upload_url}")

    response = requests.post(
        upload_url,
        files={"file": (f"client_{client_id}.pth", file)},
        data={"client_id": client_id,
              "cur_round":cur_round,
              "dataset_size":dataset_size
              }
    )

    if response.status_code == 200:
        return jsonify({"success": True, "server_response": response.json()})

    return jsonify({"success": False, "server_response": response.text}), response.status_code

if __name__ == "__main__":
    app.run(port=5000)



