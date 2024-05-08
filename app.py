from flask import Flask, request, jsonify, current_app
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

def neg_log_likelihood(params, data):
    alpha, beta = params
    return -np.sum(np.log((beta/alpha) * (data/alpha)**(beta - 1) * np.exp(-(data/alpha)**beta)))

@app.route('/')
def index():
    # Serve the HTML file from the correct path
    return current_app.send_static_file('index.html')   

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read CSV file into DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 500

        # Extract data column (assuming 'T' is the column name)
        t_values = df['TTF'].values

        # Estimate parameters for the Weibull distribution
        initial_guess = [np.mean(t_values), 0.1]
        result = minimize(neg_log_likelihood, initial_guess, args=(t_values,))
        alpha_hat, beta_hat = result.x

        # Make predictions using the ML model (Weibull reliability distribution)
        predictions = np.exp(-(t_values / alpha_hat)**beta_hat)

        # Plot the Weibull reliability distribution
        plt.figure(figsize=(8, 6))
        plt.plot(t_values, predictions, label=f'Weibull Reliability (alpha={alpha_hat:.2f}, beta={beta_hat:.2f})')
        plt.xlabel('Time to Failure (TTF)')
        plt.ylabel('Reliability (R(TTF))')
        plt.title('Weibull Reliability Distribution')
        plt.legend()
        plt.grid(True)

        # Convert plot to image and encode as base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({'image': image_base64})

# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True ,port=8080,use_reloader=False) 