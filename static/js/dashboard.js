document.addEventListener('DOMContentLoaded', function() {
    // Check if there are any active monitoring sessions
    const locationCards = document.querySelectorAll('.card');
    
    // Add click animation to cards
    locationCards.forEach(card => {
        const monitorBtn = card.querySelector('.btn-primary');
        
        monitorBtn.addEventListener('click', function() {
            // Add loading state to button
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
            this.disabled = true;
            
            // Add active class to card
            card.classList.add('border-primary');
            
            // Let the link proceed normally to the monitor page
        });
    });
    
    // If we have information about an active monitoring session, highlight that card
    const urlParams = new URLSearchParams(window.location.search);
    const activeLocation = urlParams.get('active');
    
    if (activeLocation) {
        const activeCard = document.querySelector(`[data-location="${activeLocation}"]`);
        if (activeCard) {
            activeCard.classList.add('border-primary');
            
            // Scroll to the active card
            activeCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
});