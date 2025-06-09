// Clean portfolio loader without error reporting issues
(function() {
    function updatePortfolio() {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/portfolio', true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                if (data && data.total_balance) {
                    var totalEl = document.getElementById('portfolio-total');
                    var cashEl = document.getElementById('portfolio-cash');
                    var posEl = document.getElementById('portfolio-positions');
                    
                    if (totalEl) totalEl.textContent = '$' + data.total_balance.toFixed(2);
                    if (cashEl) cashEl.textContent = '$' + data.cash_balance.toFixed(2);
                    if (posEl) posEl.textContent = data.positions.length;
                }
            }
        };
        xhr.send();
    }
    
    // Start loading
    updatePortfolio();
    setInterval(updatePortfolio, 10000);
})();