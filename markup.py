def app_intro():
    return """
    <div style='text-align: left;'>
    <h2 style='text-align: center;'>Welcome to the Lab Parameter Analysis Tool Demo</h2>
    <p style='color: red;'>Note: Please DM me via Freelancer to get the password.</p>       
    <p>Welcome to the Lab Parameter Analysis Tool Demo. This application is designed to help you analyze lab parameters in relation to surgery dates across a diverse patient base. Here's what you can do:</p>
    
    <h4>What You Can Do:</h4>
    <ul>
        <li><b>Import Data:</b> You can import or copy data from Excel files containing patient details and lab parameter reports.</li>
        <li><b>View and Filter Data:</b> You can view and filter the data to select specific patients or patient groups based on various criteria.</li>
        <li><b>Point-to-Point Analysis:</b> You can perform point-to-point analysis for specific lab parameters and patients, viewing the results both graphically and tabularly.</li>
        <li><b>Interpolated Point-to-Point Analysis:</b> You can interpolate values for lab parameters at defined times for each patient, facilitating standardized analysis.</li>
        <li><b>Generate Patient Results File:</b> You can generate a patient results file with columns representing results for each parameter and time point.</li>
    </ul>
    </div>          
    """

def how_use_intro():
    return """
    <div style='text-align: left;'>
    <h3 style='text-align: center;'>How to Use this demo</h3>
    <br>
    <h4>How to Use:</h4>
    <ul>
        <li><b>Step 1:</b> Upload or copy data from Excel files containing patient details and lab parameter reports in the "Upload Files" tab.</li>
        <li><b>Step 2:</b> In the "View and Filter Data" tab, select and filter the data based on your criteria such as lab parameters, patients, or patient groups.</li>
        <li><b>Step 3:</b> Use the "Point-to-Point Analysis" tab to perform analysis for specific lab parameters and patients. You can visualize results graphically and view tabular data.</li>
        <li><b>Step 4:</b> If needed, perform interpolated point-to-point analysis in the "Interpolated Point-to-Point Analysis" tab to standardize data.</li>
        <li><b>Step 5:</b> In the "Generate Patient Results File" tab, generate a results file with columns representing parameter values at different time points.</li>
    </ul>
    <br>
    </div>
    """
