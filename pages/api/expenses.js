import pool from '../../lib/db';

export default async function handler(req, res) {
  if (req.method === 'POST') {
    const { story, amount, date, category } = req.body;
    try {
      await pool.query(
        'INSERT INTO expense (story, amount, date, category) VALUES ($1, $2, $3, $4)',
        [story, amount, date, category]
      );
      res.status(201).json({ message: 'Expense added successfully' });
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Database error' });
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' });
  }
}
