"""Execute a notebook top-to-bottom with jupyter_client and store the outputs in place."""
import json, os, sys
from jupyter_client.manager import start_new_kernel

path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
km, kc = start_new_kernel(kernel_name="python3", cwd=os.path.dirname(os.path.abspath(path)))
errors = 0
try:
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n += 1
        src = cell["source"]
        msg_id = kc.execute("".join(src) if isinstance(src, list) else src)
        outputs = []
        while True:
            msg = kc.get_iopub_msg(timeout=900)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                break
            if t == "stream":
                outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            elif t in ("display_data", "execute_result"):
                o = {"output_type": t, "data": c["data"], "metadata": c["metadata"]}
                if t == "execute_result":
                    o["execution_count"] = n
                outputs.append(o)
            elif t == "error":
                errors += 1
                outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                "traceback": c["traceback"]})
                print("ERROR in cell", n, c["ename"], c["evalue"])
        cell["outputs"], cell["execution_count"] = outputs, n
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("executed; errors:", errors)
