var IGNORE_TAGS = { SCRIPT: 1, STYLE: 1, META: 1, NOSCRIPT: 1 };
function loop(node, p) {
  // return visible nodes
  if (!node.nodeName || node.hidden || IGNORE_TAGS[node.nodeName]) {
    return;
  }
  if (node.scrollHeight === 0 && node.scrollWidith === 0) {
    return;
  }
  let node_def = {
    t: node.nodeName,
    v: node.nodeValue || "",
    a: { id: node.id },
    c: [],
    r: p,
    d: node.getBoundingClientRect ? node.getBoundingClientRect() : null,
    f: node === document.activeElement
  };
  if (node.attributes) {
    for (var i = 0; i < node.attributes.length; i++) {
      node_def.a[node.attributes[i].name] = node.attributes[i].value;
    }
  }
  if (node instanceof HTMLInputElement) {
    var inputType = node.type;
    if (inputType === "checkbox" || inputType === "radio") {
      node_def.v = node.checked;
    } else {
      node_def.v = node.value;
    }
  } else if (node instanceof HTMLTextAreaElement) {
    node_def.v = node.value;
  }
  var nodes = node.childNodes || [];
  var ci = 0;
  var hasText = "";
  for (var i = 0; i < nodes.length; i++) {
    let n = nodes[i];
    if (!n) {
      continue;
    }
    if (n.nodeName === "#text") {
      hasText += n.nodeValue;
    } else if (n.childNodes.length > 0) {
      let c = loop(n, ci.toString());
      if (c) node_def.c.push(c);
      ci += 1;
    }
  }
  hasText = hasText.trim().length > 0;
  if (hasText) {
    node_def.v = node.innerText;
  }
  return node_def;
}

window.core = {
  elementClick: function(element) {
    if (typeof element === "string") {
      element = document.querySelector(element);
    }
    return element && element.click && (element.click() || true);
  },
  getDOMInfo: function() {
    window.core.previousDOMInfo = loop(document.body, "html");
    return window.core.previousDOMInfo;
  },
  recordedErrors: [],
  getErrors: function() {
    ret = window.core.recordedErrors;
    window.core.recordedErrors = []
    return ret;
  }
};

window.addEventListener('error', function(event) {
  window.core.recordedErrors.push(event);
});
