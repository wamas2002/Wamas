
// Visual Enhancement Effects for Modern Trading Dashboard

document.addEventListener('DOMContentLoaded', function() {
    initializeVisualEffects();
    setupInteractiveElements();
    startAnimationSystem();
});

function initializeVisualEffects() {
    // Add intersection observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe all cards and metrics
    document.querySelectorAll('.metric-card, .card, .mini-widget').forEach(el => {
        observer.observe(el);
    });
}

function setupInteractiveElements() {
    // Enhanced hover effects for metric cards
    document.querySelectorAll('.metric-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.03)';
            this.style.boxShadow = '0 20px 60px rgba(0, 0, 0, 0.3), 0 0 40px rgba(79, 139, 255, 0.4)';
            
            // Add ripple effect
            const ripple = document.createElement('div');
            ripple.className = 'ripple-effect';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });

    // Interactive buttons with enhanced feedback
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const ripple = document.createElement('span');
            ripple.className = 'btn-ripple';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

function startAnimationSystem() {
    // Staggered loading animations
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });

    // Dynamic loading indicators
    updateLoadingAnimations();
    setInterval(updateLoadingAnimations, 2000);
    
    // Smooth number transitions
    animateNumbers();
}

function updateLoadingAnimations() {
    document.querySelectorAll('.loading').forEach(loader => {
        loader.style.animation = 'none';
        loader.offsetHeight; // Trigger reflow
        loader.style.animation = 'spin 1s linear infinite, shimmer 2s ease-in-out infinite';
    });
}

function animateNumbers() {
    document.querySelectorAll('.metric-value').forEach(element => {
        const text = element.textContent;
        const number = parseFloat(text.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(number)) {
            animateValue(element, 0, number, 1000);
        }
    });
}

function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const prefix = element.textContent.replace(/[\d.-]/g, '');
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (end - start) * easeOutQuart(progress);
        element.textContent = prefix + current.toFixed(2);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
}

// Add CSS for new effects
const style = document.createElement('style');
style.textContent = `
    .ripple-effect {
        position: absolute;
        border-radius: 50%;
        background: rgba(79, 139, 255, 0.3);
        transform: scale(0);
        animation: rippleAnimation 0.6s ease-out;
        pointer-events: none;
        width: 100px;
        height: 100px;
        left: 50%;
        top: 50%;
        margin-left: -50px;
        margin-top: -50px;
    }

    @keyframes rippleAnimation {
        to {
            transform: scale(2);
            opacity: 0;
        }
    }

    .btn-ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: btnRipple 0.6s ease-out;
        pointer-events: none;
        width: 20px;
        height: 20px;
        margin-left: -10px;
        margin-top: -10px;
    }

    @keyframes btnRipple {
        to {
            transform: scale(3);
            opacity: 0;
        }
    }

    .animate-in {
        animation: slideInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-value {
        background: linear-gradient(45deg, #4f8bff, #9c27b0, #00d395);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 3s ease-in-out infinite;
    }

    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
`;
document.head.appendChild(style);
