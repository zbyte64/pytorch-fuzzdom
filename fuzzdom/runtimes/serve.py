from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

from fuzzdom.state import MiniWoBGraphState, DomInfo, Fields
from fuzzdom.vec_env import state_to_vector
from fuzzdom.data import TupleBatch
from fuzzdom.train import rdncrawl
from fuzzdom.factory_resolver import FactoryResolver
from fuzzdom.domx import miniwob_to_dominfo

app = Flask(__name__, static_url_path="")
CORS(app, send_wild_card=True)

resolver = FactoryResolver(rdncrawl)


@app.route("/init.js")
def init_js():
    return send_from_directory("/code/fuzzdom/js", "init.js")


@app.route("/heatmap.js")
def heatmap_js():
    return send_from_directory("/code/fuzzdom/js", "heatmap.js")


@app.route("/heatmap/<key>/<value>", methods=["POST"])
def heatmap(key, value):
    content = request.json
    if "col" not in content:
        content = miniwob_to_dominfo(content)
    dom_info = DomInfo(**content)
    fields = Fields({key: value})
    state = MiniWoBGraphState(
        utterance="",
        fields=fields,
        dom_info=dom_info,
        screenshot=None,
        logs={"errors": []},
    )

    actor_critic = resolver["actor_critic"]
    filter_leaves = resolver["filter_leaves"]
    obs = state_to_vector(state, filter_leaves, num_of_actions=2)
    batch_data = tuple(
        (b.to(resolver["device"]) for b in TupleBatch.from_data_list([obs]))
    )
    critic_value, action_votes, _ = actor_critic.base(
        batch_data, rnn_hxs=None, masks=None
    )
    votes, action_batch_idx = action_votes
    votes = votes.view(-1)
    (dom, objectives, obj_projection, leaves, actions, *_) = obs
    # app.logger.debug(str(votes))
    # app.logger.debug(str(actions.combinations))
    combo = lambda combination_idx: {
        k: v for i, c in actions.combinations[combination_idx] for k, v in c.items()
    }
    return jsonify(
        [
            (
                dom_info.ref[combo(combination_idx)["dom_idx"]],
                combo(combination_idx)["action_idx"],
                v,
            )
            for combination_idx, v in enumerate(votes.tolist())
        ]
    )


if __name__ == "__main__":
    args = resolver["args"]
    args.load_autoencoder, args.load_actor, args.load_critic = True, True, True
    resolver["actor_critic"].eval()
    app.run("0.0.0.0", debug=True)
