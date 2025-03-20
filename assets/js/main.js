document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Add active class to current nav item
    const currentLocation = location.href;
    const menuItems = document.querySelectorAll('.nav-menu a');
    menuItems.forEach(item => {
        if(item.href === currentLocation) {
            item.classList.add('active');
        }
    });

    fetch('assets/text/readme-example.txt')
        .then(response => response.text())
        .then(data => {
            document.getElementById('readme-content').innerText = data;
        })
        .catch(error => console.error('Error loading text:', error));
    
});

function copyCode() {
    const codeBlock = document.getElementById('codeBlock').innerText;

    navigator.clipboard.writeText(codeBlock)
      .then(() => alert('Code copied to clipboard!'))
      .catch(err => alert('Failed to copy code: ' + err));
}
