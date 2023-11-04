
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



// JavaScript for handling microphone input


function sendTranscriptToServer(transcript) {
  // Define the URL of your Django view for processing
  const url = './home/views.py/';

  // Create a FormData object to send the transcript as POST data
  const formData = new FormData();
  formData.append('transcript', transcript);

  // Send a POST request to the server
  fetch(url, {
    method: 'POST',
    body: formData,
    headers: {
      // You may need to set additional headers if required by your Django view
    },
  })
    .then(response => response.json())  // Assuming the response from the server is JSON
    .then(data => {
      // Handle the response from the server if needed
      console.log('Server response:', data);
    })
    .catch(error => {
      // Handle any errors that occur during the fetch
      console.error('Fetch error:', error);
    });
}