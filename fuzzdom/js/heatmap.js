function view_activity(key, value) {
  var domInfo = core.getDOMInfo();
  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://localhost:5000/heatmap/'+key+'/'+value, true);
  xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var json = JSON.parse(xhr.responseText);
        json.map(function(x) {
          ref = x[0]
          action_id = x[1]
          v = x[2]
          window.core.previousDOMInfo[ref].setAttribute('data-action-'+action_id, v);
        });
    }
  };
  xhr.send(JSON.stringify(domInfo));
}
