import axios from 'axios';

const exerciseClient = axios.create({
  baseURL: 'https://exercisedb.p.rapidapi.com',
  headers: {
    'x-rapidapi-key': '406c2277c7msh469fe72e9b76a5dp12004cjsna55ea35dcc5b',
    'x-rapidapi-host': 'exercisedb.p.rapidapi.com',
    'Content-Type': 'application/json',
  },
});

export default exerciseClient;
