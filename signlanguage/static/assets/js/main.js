
window.onload = switchASL();
window.onload = switchNSL();
document.getElementById('asl').style.display == 'none';
document.getElementById('nsl').style.display == 'none';
function switchASL() {
  if (document.getElementById('asl').style.display == 'none') {
    document.getElementById('asl').style.display = 'block';
    document.getElementById('nsl').style.display = 'none';
  }
  else {
    document.getElementById('asl').style.display = 'none';
  }
}
function switchNSL() {
  if (document.getElementById('nsl').style.display == 'none') {
    document.getElementById('nsl').style.display = 'block';
    document.getElementById('asl').style.display = 'none';
  }
  else {
    document.getElementById('nsl').style.display = 'none';
  }
}

