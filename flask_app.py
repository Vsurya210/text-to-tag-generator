# flask_app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from job_api import fetch_job_descriptions
import os

# import your NLP tagger
from text_to_tag import TextToTag, load_tags_from_csv, DEFAULT_TAGS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL = None


def build_model(tags_path=None):
    global MODEL

    if tags_path and os.path.exists(tags_path):
        try:
            tags = load_tags_from_csv(tags_path)
        except:
            tags = DEFAULT_TAGS
    else:
        tags = DEFAULT_TAGS

    MODEL = TextToTag(tags)
    return MODEL


# Build automatically if tags.csv exists
if os.path.exists("tags.csv"):
    build_model("tags.csv")
else:
    build_model()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", tags=None, top_k=5, min_score=0.03, sample_text="Enter text here...")


@app.route("/build", methods=["POST"])
def build():
    tags_path = "tags.csv" if os.path.exists("tags.csv") else None
    try:
        model = build_model(tags_path)
        return jsonify({"status": "ok", "n_tags": len(model.tag_names)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    global MODEL

    # HTML form submission
    if request.form.get("text"):
        text = request.form.get("text")
        top_k = int(request.form.get("top_k", 5))
        min_score = float(request.form.get("min_score", 0.03))

        preds = MODEL.predict(text, top_k=top_k, min_score=min_score)
        formatted = [{"tag": t, "score": s} for t, s in preds]

        return render_template("index.html",
                               tags=formatted,
                               top_k=top_k,
                               min_score=min_score,
                               sample_text=text)

    # JSON API
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    top_k = int(data.get("top_k", 5))
    min_score = float(data.get("min_score", 0.03))

    preds = MODEL.predict(text, top_k=top_k, min_score=min_score)
    return jsonify({"tags": [{"tag": t, "score": s} for t, s in preds]})

@app.route("/job-tags", methods=["GET"])
def job_tags():
    global MODEL

    query = request.args.get("query", "python developer")
    limit = int(request.args.get("limit", 5))

    job_texts = fetch_job_descriptions(query=query, limit=limit)

    results = []
    for text in job_texts:
        tags = MODEL.predict(text, top_k=5)
        results.append({
            "job_excerpt": text[:200],
            "tags": [{"tag": t, "score": s} for t, s in tags]
        })

    return jsonify(results)



if __name__ == "__main__":
    app.run(debug=True, port=5000)
