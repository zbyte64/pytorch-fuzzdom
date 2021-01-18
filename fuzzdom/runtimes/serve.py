from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

from fuzzdom.state import MiniWoBGraphState, DomInfo, Fields
from fuzzdom.vec_env import state_to_vector
from fuzzdom.data import TupleBatch
from fuzzdom.train import rdncrawl
from fuzzdom.factory_resolver import FactoryResolver

app = Flask(__name__, static_url_path="")
CORS(app)

resolver = FactoryResolver(rdncrawl)


@app.route("/init.js")
def init_js():
    return send_from_directory("/code/fuzzdom/js", "init.js")


@app.route("/heatmap.js")
def init_js():
    return send_from_directory("/code/fuzzdom/js", "heatmap.js")


@app.route("/heatmap/<key>/<value>")
def heatmap(key, value):
    content = request.json
    dom_info = DomInfo(**content)
    fields = Fields({key: value})
    state = MiniWoBGraphState(
        utterance="", fields=fields, dom_info=dom_info, screenshot=None, logs=dict()
    )

    actor_critic = resolver["actor_critic"]
    filter_leaves = resolver["filter_leaves"]
    obs = state_to_vector(state, filter_leaves)
    batch_data = TupleBatch.from_data_list([obs])
    critic_value, action_votes, _ = actor_critic(batch_data)
    votes, action_batch_idx = action_votes
    (dom, objectives, obj_projection, leaves, actions, *_) = obs
    return jsonify(
        [
            (
                obs.dom_info.ref[actions.combinations[combination_idx]["dom_idx"]],
                actions.combinations[combination_idx]["action_idx"],
                v,
            )
            for combination_idx, v in zip(action_batch_idx, critic_value)
        ]
    )


if __name__ == "__main__":
    args = resolver["args"]
    args.load_autoencoder, args.load_actor, args.load_critic = True, True, True
    resolver["actor_critic"].eval()
    app.run("0.0.0.0", debug=True)
