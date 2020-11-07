var IGNORE_TAGS = { SCRIPT: 1, STYLE: 1, META: 1, NOSCRIPT: 1 };
var DOM_COUNTER = 0;
function loop(domInfo, node, depth, p) {
  // return visible nodes
  if (
    !node.nodeName ||
    !node.click ||
    node.hidden ||
    IGNORE_TAGS[node.nodeName]
  ) {
    return;
  }
  if (node.scrollHeight === 0 && node.scrollWidith === 0) {
    return;
  }
  let d = node.getBoundingClientRect ? node.getBoundingClientRect() : null;
  if (!d) {
    return;
  }
  if (node._ref === undefined) {
    node._ref = DOM_COUNTER;
    DOM_COUNTER += 1;
  }
  let node_def = {
    ref: node._ref,
    tag: node.nodeName,
    value: node.nodeValue || "",
    text: "",
    focused: node === document.activeElement,
    classes: node.className,
    rx: (d.left - p && p.left) || 0,
    ry: (d.top - p && p.top) || 0,
    height: d.height,
    width: d.width,
    top: d.top,
    left: d.left,
    depth: depth,
    tampered: node._tampered === true,
    n_children: 0,
  };
  if (node.type) {
    node_def.tag = node_def.tag + "_" + node.type;
  } else if (node.hasAttribute("role")) {
    node_def.tag = node_def.tag + "_" + node.getAttribute("role");
  }
  if (node instanceof HTMLInputElement) {
    var inputType = node.type;
    if (inputType === "checkbox" || inputType === "radio") {
      node_def.value = node.checked;
    } else {
      node_def.value = node.value;
    }
  } else if (node instanceof HTMLTextAreaElement) {
    node_def.value = node.value;
  } else if (node.hasAttribute("aria-checked")) {
    node_def.value = node.getAttribute("aria-checked");
  }

  core.previousDOMInfo[node_def["ref"]] = node;

  var nodes = node.childNodes || [];
  var hasText = "";
  var childIndices = [];
  for (var i = 0; i < nodes.length; i++) {
    let n = nodes[i];
    if (!n) {
      continue;
    }
    if (n.nodeName === "#text") {
      hasText += n.nodeValue;
    } else if (n.childNodes.length > 0) {
      //only care if the node itself has text as well
      let c = loop(domInfo, n, depth + 1, d);
      if (c !== undefined) {
        childIndices.push(c);
        node_def.n_children += 1;
      }
    }
  }
  hasText = hasText.trim().length > 0;
  if (hasText) {
    node_def.text = node.innerText;
  } else if (node_def.n_children === 0) {
    node_def.text = node.innerText;
  }
  ["aria-label", "alt", "label", "placeholder", "title"].map(function (x) {
    if (node.hasAttribute(x)) node_def.text += " " + node.getAttribute(x);
  });
  var index = domInfo.ref.length;
  Object.keys(node_def).map(function (key) {
    domInfo[key].push(node_def[key]);
  });
  childIndices.map(function (c) {
    domInfo.row.push(index);
    domInfo.col.push(c);
  });
  return index;
}

prior_core = window.core;
window.core = Object.assign(
  {
    elementClick: function (element) {
      if (typeof element === "string" || typeof element === "number") {
        element = window.core.previousDOMInfo[element];
      }
      if (element) element._tampered = true;
      return element && element.click && (element.click() || true);
    },
    elementType: function (element, text) {
      if (typeof element === "string" || typeof element === "number") {
        element = window.core.previousDOMInfo[element];
      }
      if (element) {
        element._tampered = true;
        element.click();
        element.focus();
        element.value = text;
        return true;
      }
    },
    getDOMInfo: function () {
      var domInfo = {
        ref: [],
        tag: [],
        value: [],
        text: [],
        focused: [],
        classes: [],
        rx: [],
        ry: [],
        height: [],
        width: [],
        top: [],
        left: [],
        depth: [],
        tampered: [],
        n_children: [],
        row: [],
        col: [],
      };
      window.core.previousDOMInfo = {};
      loop(domInfo, document.body, 0);
      return domInfo;
    },
    logs: {},
    getLogs: function () {
      ret = window.core.logs;
      window.core.logs = {};
      Object.keys(ret).map(function (x) {
        window.core.logs[x] = [];
      });
      return ret;
    },
  },
  prior_core
);
if (prior_core !== undefined) {
  Object.assign(prior_core, window.core)
}

window.addEventListener("error", function (event) {
  window.core.logs["windowError"].push(event);
});

function bindConsoleLog(attr) {
  fn = console[attr];
  f = fn.bind(console);
  window.core.logs[attr] = [];
  console[attr] = function () {
    window.core.logs[attr].push(Array.from(arguments));
    f.apply(console, arguments);
  };
}

if (console.log) bindConsoleLog("log");
if (console.error) bindConsoleLog("error");
