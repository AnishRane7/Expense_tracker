const handleSubmit = async (e) => {
    e.preventDefault();
  
    // Step 1: Call the Colab API to get the category
    const categoryResponse = await fetch('https://xyz.ngrok.io/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story }),
    });
  
    const { category } = await categoryResponse.json();
  
    // Step 2: Use the category in your database request
    const response = await fetch('/api/expenses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story, amount, date, category }),
    });
  
    if (response.ok) {
      // Handle success, e.g., clear form or show success message
    }
  };
  