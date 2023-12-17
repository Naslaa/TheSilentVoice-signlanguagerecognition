
window.onload = switchASL();
window.onload = switchNSL();
document.getElementById('asl').style.display == 'none';
document.getElementById('nsl').style.display == 'none';
function switchASL() {
  console.log("clicked asl")
  if (document.getElementById('asl').style.display == 'none') {
    document.getElementById('asl').style.display = 'block';
    document.getElementById('nsl').style.display = 'none';
  }
  else {
    document.getElementById('asl').style.display = 'none';
  }
}
function switchNSL() {
  console.log("clicked nsl")
  if (document.getElementById('nsl').style.display == 'none') {
    document.getElementById('nsl').style.display = 'block';
    document.getElementById('asl').style.display = 'none';
  }
  else {
    document.getElementById('nsl').style.display = 'none';
  }
}

