/**
 * Dashboard Analytics Module
 * Handles data visualization and user interactions for the analytics dashboard
 */

// Configuration object
const dashboardConfig = {
    apiEndpoint: 'https://api.example.com/analytics',
    refreshInterval: 60000, // milliseconds
    defaultDateRange: 30, // days
    colorScheme: {
        primary: '#4285F4',
        secondary: '#34A853',
        warning: '#FBBC05',
        error: '#EA4335',
        background: '#FFFFFF'
    }
};

// Cache for API responses
let dataCache = {};

/**
 * Fetches analytics data from the API
 * @param {string} metric - The metric to fetch (e.g., 'users', 'revenue', 'engagement')
 * @param {number} days - Number of days to include in the analysis
 * @returns {Promise<Object>} - Promise resolving to the data object
 */
async function fetchAnalyticsData(metric, days = dashboardConfig.defaultDateRange) {
    const cacheKey = `${metric}_${days}`;
    
    // Return cached data if available and not expired
    if (dataCache[cacheKey] && dataCache[cacheKey].timestamp > Date.now() - dashboardConfig.refreshInterval) {
        console.log(`Using cached data for ${cacheKey}`);
        return dataCache[cacheKey].data;
    }
    
    try {
        const response = await fetch(`${dashboardConfig.apiEndpoint}/${metric}?days=${days}`);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save to cache
        dataCache[cacheKey] = {
            timestamp: Date.now(),
            data: data
        };
        
        return data;
    } catch (error) {
        console.error('Failed to fetch analytics data:', error);
        throw error;
    }
}

/**
 * Creates a chart using the provided data
 * @param {string} elementId - The ID of the DOM element to render the chart in
 * @param {Object} data - The data to visualize
 * @param {string} chartType - The type of chart to create (bar, line, pie)
 */
function createChart(elementId, data, chartType = 'line') {
    const element = document.getElementById(elementId);
    
    if (!element) {
        console.error(`Element with ID ${elementId} not found`);
        return;
    }
    
    // Clear any existing chart
    element.innerHTML = '';
    
    // Basic validation of data
    if (!data || !data.labels || !data.values || data.labels.length !== data.values.length) {
        console.error('Invalid data structure for chart');
        element.innerHTML = '<div class="error-message">Invalid data format</div>';
        return;
    }
    
    // Chart would be created here using a charting library
    // This is a placeholder implementation
    console.log(`Creating ${chartType} chart in ${elementId} with ${data.labels.length} data points`);
    
    // Mock chart creation
    let chartHtml = `<div class="chart-container chart-${chartType}">`;
    chartHtml += `<h3>${data.title || 'Analytics Chart'}</h3>`;
    chartHtml += '<div class="chart-visualization">';
    
    // Simple visualization placeholder
    const maxValue = Math.max(...data.values);
    for (let i = 0; i < data.labels.length; i++) {
        const percentage = (data.values[i] / maxValue) * 100;
        chartHtml += `
            <div class="chart-item">
                <div class="chart-bar" style="height: ${percentage}%; background-color: ${dashboardConfig.colorScheme.primary}"></div>
                <div class="chart-label">${data.labels[i]}</div>
            </div>
        `;
    }
    
    chartHtml += '</div></div>';
    element.innerHTML = chartHtml;
}

/**
 * Initializes the dashboard with default metrics
 */
function initializeDashboard() {
    console.log('Initializing analytics dashboard...');
    
    // Set up event listeners for dashboard controls
    document.querySelectorAll('.date-range-selector').forEach(selector => {
        selector.addEventListener('change', handleDateRangeChange);
    });
    
    document.querySelectorAll('.refresh-button').forEach(button => {
        button.addEventListener('click', refreshDashboard);
    });
    
    // Initial data load
    loadDashboardData();
    
    // Set up auto-refresh
    setInterval(refreshDashboard, dashboardConfig.refreshInterval);
    
    console.log('Dashboard initialization complete');
}

/**
 * Loads all dashboard data
 */
async function loadDashboardData() {
    try {
        // Show loading state
        document.querySelectorAll('.chart-container').forEach(container => {
            container.classList.add('loading');
        });
        
        // Fetch different metrics in parallel
        const [userData, revenueData, engagementData] = await Promise.all([
            fetchAnalyticsData('users'),
            fetchAnalyticsData('revenue'),
            fetchAnalyticsData('engagement')
        ]);
        
        // Create charts with the retrieved data
        createChart('users-chart', userData, 'line');
        createChart('revenue-chart', revenueData, 'bar');
        createChart('engagement-chart', engagementData, 'pie');
        
        // Update last refresh timestamp
        document.getElementById('last-refresh-time').textContent = new Date().toLocaleTimeString();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        document.getElementById('dashboard-error').textContent = 'Failed to load dashboard data. Please try again.';
    } finally {
        // Remove loading state
        document.querySelectorAll('.chart-container').forEach(container => {
            container.classList.remove('loading');
        });
    }
}

/**
 * Handles change of date range selector
 * @param {Event} event - The change event
 */
function handleDateRangeChange(event) {
    const days = parseInt(event.target.value, 10);
    console.log(`Date range changed to ${days} days`);
    
    // Clear cache to force fresh data load
    dataCache = {};
    
    // Update config
    dashboardConfig.defaultDateRange = days;
    
    // Reload data with new range
    loadDashboardData();
}

/**
 * Refreshes all dashboard data
 */
function refreshDashboard() {
    console.log('Manually refreshing dashboard data...');
    
    // Clear cache to force fresh data load
    dataCache = {};
    
    // Reload all data
    loadDashboardData();
}

// Export public API
window.AnalyticsDashboard = {
    initialize: initializeDashboard,
    refresh: refreshDashboard,
    createChart: createChart,
    fetchData: fetchAnalyticsData
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeDashboard);