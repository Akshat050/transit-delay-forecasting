# Contributing to DelaySenseAI

Thank you for your interest in contributing to DelaySenseAI! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/delaysenseai.git
   cd delaysenseai
   ```

### 2. Set Up Development Environment
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

### 3. Make Your Changes
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   python main.py  # Test the main pipeline
   streamlit run dashboard.py  # Test the dashboard
   ```

3. Ensure code quality:
   - Follow PEP 8 style guidelines
   - Add docstrings to new functions
   - Update tests if applicable

### 4. Commit and Push
1. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### 5. Create a Pull Request
1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select the main branch as the base
4. Add a description of your changes
5. Submit the PR

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add type hints where appropriate
- Keep functions focused and concise

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions and classes
- Include examples for new functionality

### Testing
- Test your changes with sample GTFS data
- Ensure the dashboard loads without errors
- Verify that new features work as expected

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: Improve memory usage and processing speed
- **Error Handling**: Add robust error handling for edge cases
- **Testing**: Add unit tests and integration tests
- **Documentation**: Improve code documentation and user guides

### Medium Priority
- **New Features**: Add new visualization types or analysis tools
- **UI Improvements**: Enhance the Streamlit dashboard interface
- **Data Processing**: Support for additional GTFS data formats
- **Model Improvements**: Experiment with new ML algorithms

### Low Priority
- **Code Refactoring**: Improve code structure and organization
- **Performance Monitoring**: Add logging and performance metrics
- **Internationalization**: Support for multiple languages

## 🐛 Reporting Issues

When reporting issues, please include:
1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Step-by-step instructions
3. **Expected vs Actual Behavior**: What you expected vs what happened
4. **Environment**: Python version, OS, and package versions
5. **Sample Data**: If possible, provide sample GTFS data that causes the issue

## 💡 Feature Requests

For feature requests, please:
1. **Describe the Feature**: What would you like to see?
2. **Use Case**: How would this feature be used?
3. **Benefits**: What problems would it solve?
4. **Mockups**: If applicable, provide mockups or examples

## 📞 Getting Help

- **GitHub Issues**: Use the Issues tab for bugs and feature requests
- **GitHub Discussions**: Use Discussions for questions and general discussion
- **Email**: Contact the maintainers directly for urgent issues

## 🏷️ Commit Message Format

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: change description`
- `Remove: feature removal description`
- `Docs: documentation update`
- `Style: code formatting changes`

## 📄 License

By contributing to DelaySenseAI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- The README.md file
- Release notes
- Project documentation

Thank you for contributing to making transit systems better! 🚌 