// server.js or app.js
const express = require('express');
const app = express();
const { Client } = require('pg');

app.get('/', (req, res)=>{
    res.send('hello')
})

app.get('/api/data/hsi', (req, res)=>{
    const client = new Client({
    user: 'postgres',
    host: 'localhost',
    database: 'finance',
    password: '1234',
    port: 5432,
    });
    client.connect();
    client.query('SELECT * FROM stock_performance_hsi', (err, result) => {
        if (err) {
            console.error(err);
            res.status(500).send('ERROR fetching data')
        }
        res.json(result.rows)
        console.log(result.rows);
        client.end();
    });
})

app.get('/api/data/sse', (req, res)=>{
    const client = new Client({
    user: 'postgres',
    host: 'localhost',
    database: 'finance',
    password: '1234',
    port: 5432,
    });
    client.connect();
    client.query('SELECT * FROM stock_performance_sse', (err, result) => {
        if (err) {
            console.error(err);
            res.status(500).send('ERROR fetching data')
        }
        res.json(result.rows)
        console.log(result.rows);
        client.end();
    });
})

app.get('/api/data/nasdaq', (req, res)=>{
    const client = new Client({
    user: 'postgres',
    host: 'localhost',
    database: 'finance',
    password: '1234',
    port: 5432,
    });
    client.connect();
    client.query('SELECT * FROM stock_performance_nasdaq', (err, result) => {
        if (err) {
            console.error(err);
            res.status(500).send('ERROR fetching data')
        }
        res.json(result.rows)
        console.log(result.rows);
        client.end();
    });
})

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});