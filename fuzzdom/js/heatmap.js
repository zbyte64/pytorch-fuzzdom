function view_activity(key, value) {
  var actionMap = ['click', 'type', 'copy', 'paste', 'sleep'];
  var domInfo = core.getDOMInfo();
  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://localhost:5000/heatmap/'+key+'/'+value, true);
  xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
  xhr.setRequestHeader('Origin', '*');
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var json = JSON.parse(xhr.responseText);
        json.map(function(x) {
          ref = x[0]
          action_id = x[1]
          v = x[2]
          window.core.previousDOMInfo[ref].setAttribute('data-fuzzdom-'+actionMap[action_id], v);
        });
    }
  };
  xhr.send(JSON.stringify(domInfo));
}
/*
script = document.createElement("script");
script.type = "text/javascript"
script.src = "http://localhost:5000/init.js"
document.body.appendChild(script);

script = document.createElement("script");
script.type = "text/javascript"
script.src = "http://localhost:5000/heatmap.js"
document.body.appendChild(script);
*/
