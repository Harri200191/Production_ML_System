import React from 'react';
import './searchBar.scss';

export const SearchBar = () => {
  return (
    <div className="search-container">
      <h1 className="tool-name">Vector Search Engine</h1>
      <form className="search-box">
        <input type="text" placeholder="Search..." />
        <button type="submit">Search</button>
      </form>
    </div>
  );
};
