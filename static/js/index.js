window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Icon effects: float + hover flutter + glow
function setupIconEffects() {
    const icon = document.querySelector('.publication-icon');
    if (!icon) return;
    icon.addEventListener('mouseenter', () => {
        icon.classList.add('spinning');
    });
    icon.addEventListener('mouseleave', () => {
        icon.classList.remove('spinning');
    });
}

// Reading progress bar
function setupReadingProgress() {
    const bar = document.getElementById('reading-progress');
    if (!bar) return;
    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.width = (scrollTop / docHeight * 100) + '%';
    }, { passive: true });
}

// 3D tilt on stat cards
function setupTiltEffect() {
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transition = 'transform 0.1s ease, box-shadow 0.1s ease';
        });
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width  - 0.5;
            const y = (e.clientY - rect.top)  / rect.height - 0.5;
            card.style.transform = `perspective(600px) rotateY(${x * 16}deg) rotateX(${-y * 16}deg) translateY(-6px)`;
            card.style.boxShadow = `${-x * 12}px ${y * 12}px 24px rgba(37,99,235,0.15)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transition = 'transform 0.4s ease, box-shadow 0.4s ease';
            card.style.transform = '';
            card.style.boxShadow = '';
        });
    });
}

// Stat card count-up animation
function setupCountUp() {
    const statNumbers = document.querySelectorAll('.stat-number[data-target]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.dataset.target);
            const prefix = el.dataset.prefix || '';
            const suffix = el.dataset.suffix || '';
            const duration = 1200;
            const startTime = performance.now();

            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
                const current = Math.round(eased * target);
                el.textContent = prefix + current + suffix;
                if (progress < 1) requestAnimationFrame(update);
            }
            requestAnimationFrame(update);
            observer.unobserve(el);
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(el => observer.observe(el));
}

// Scroll reveal animation
function setupScrollReveal() {
    const elements = document.querySelectorAll('.hero:not(:first-of-type), .section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.03 });

    elements.forEach(el => {
        el.classList.add('scroll-reveal');
        observer.observe(el);
    });
}

// Video autoplay when in view (play only, no pause, avoids conflicts)
function setupVideoAutoplay() {
    const videos = document.querySelectorAll('video[autoplay]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.play().catch(() => {});
            }
        });
    }, { threshold: 0.1 });

    videos.forEach(video => observer.observe(video));
}

// Sync BEV video height to match paired Video height
function setupPlanningHeights() {
    const grid = document.querySelectorAll('.planning-pair');
    // Pairs are: [vid0, bev1, vid2, bev3, vid4, bev5, ...]
    // In each grid, odd-index .planning-pair cells are BEV
    const pairs = [];
    for (let i = 0; i < grid.length; i += 2) {
        pairs.push({ vid: grid[i].querySelector('video'), bev: grid[i+1] && grid[i+1].querySelector('video') });
    }

    function syncHeights() {
        pairs.forEach(({ vid, bev }) => {
            if (!vid || !bev) return;
            const h = vid.offsetHeight;
            if (h > 0) bev.style.height = h + 'px';
        });
    }

    pairs.forEach(({ vid }) => {
        if (!vid) return;
        vid.addEventListener('loadedmetadata', syncHeights);
        vid.addEventListener('resize', syncHeights);
    });
    window.addEventListener('resize', syncHeights);
    setTimeout(syncHeights, 500);
}

document.addEventListener('DOMContentLoaded', function() {
    setupIconEffects();
    setupReadingProgress();
    setupTiltEffect();
    setupCountUp();
    setupScrollReveal();
    setupVideoAutoplay();
    setupPlanningHeights();
});
