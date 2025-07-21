// Password protection for review site
// Add this script to the head of all review site pages

(function() {
    // Only run on review site
    const isReviewSite = window.location.hostname.includes('review.rob-blog.co.uk') || 
                        window.location.hostname === 'localhost' || 
                        window.location.hostname === '127.0.0.1';
    
    if (!isReviewSite) {
        return; // Don't run protection on main site
    }
    
    // Check if user has access
    const hasAccess = sessionStorage.getItem('reviewAccess') === 'granted';
    
    // Pages that don't need protection
    const publicPages = ['/index.html', '/', '/staging-password-protection.html'];
    const currentPage = window.location.pathname;
    
    // If user doesn't have access and isn't on a public page, redirect to password page
    if (!hasAccess && !publicPages.includes(currentPage)) {
        window.location.href = '/index.html';
        return;
    }
    
    // Add a logout function for convenience
    window.reviewLogout = function() {
        sessionStorage.removeItem('reviewAccess');
        window.location.href = '/index.html';
    };
    
    // Add logout button to header if user has access
    if (hasAccess && document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            addLogoutButton();
        });
    } else if (hasAccess) {
        addLogoutButton();
    }
    
    function addLogoutButton() {
        const header = document.querySelector('.site-header .trigger');
        if (header && !document.getElementById('review-logout')) {
            const logoutBtn = document.createElement('button');
            logoutBtn.id = 'review-logout';
            logoutBtn.innerHTML = 'Logout';
            logoutBtn.className = 'theme-toggle'; // Use same styling as theme toggle
            logoutBtn.style.marginLeft = '8px';
            logoutBtn.onclick = window.reviewLogout;
            logoutBtn.title = 'Logout from review site';
            header.appendChild(logoutBtn);
        }
    }
})();