require('dotenv').config();
const plaid = require('plaid');
const moment = require('moment');

// 1. Initialize the client
const client = new plaid.PlaidApi(
  new plaid.Configuration({
    basePath: plaid.PlaidEnvironments[process.env.PLAID_ENV],
    baseOptions: {
      headers: {
        'PLAID-CLIENT-ID': process.env.PLAID_CLIENT_ID,
        'PLAID-SECRET': process.env.PLAID_SECRET,
      },
    },
  })
);

// 2. Create Link Token
async function createLinkToken() {
  const response = await client.linkTokenCreate({
    user: { client_user_id: 'user-123' },
    client_name: 'Finance Agent',
    products: ['transactions'],
    country_codes: ['US'],
    language: 'en',
  });
  console.log('Link Token:', response.data.link_token);
  return response.data.link_token;
}

// 3. Exchange public token for access token
async function exchangePublicToken(publicToken) {
  const response = await client.itemPublicTokenExchange({
    public_token: publicToken,
  });
  console.log('Access Token:', response.data.access_token);
  return response.data.access_token;
}

// 4. Get Accounts
async function getAccounts(accessToken) {
  const response = await client.accountsGet({ access_token: accessToken });
  console.log('Accounts:', response.data.accounts);
  return response.data.accounts;
}

// 5. Get Transactions (last 6 months)
async function getTransactions(accessToken) {
  const startDate = moment().subtract(6, 'months').format('YYYY-MM-DD');
  const endDate = moment().format('YYYY-MM-DD');

  const response = await client.transactionsGet({
    access_token: accessToken,
    start_date: startDate,
    end_date: endDate,
    options: { count: 100 },
  });

  console.log('Transactions:', response.data.transactions);
  return response.data.transactions;
}

// 🔁 Example Usage Flow
(async () => {
  try {
    const linkToken = await createLinkToken();
    console.log('Use this token in Plaid Link frontend.');

    // Simulate getting a public token from frontend (in reality this is returned after user logs in)
    const PUBLIC_TOKEN = 'sandbox-simulator-token'; // Replace with real token after linking
    const accessToken = await exchangePublicToken(PUBLIC_TOKEN);

    await getAccounts(accessToken);
    await getTransactions(accessToken);
  } catch (err) {
    console.error(err.response?.data || err);
  }
})();
