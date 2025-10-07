# Contributing to CESAROPS

Thank you for your interest in contributing to CESAROPS! This document provides guidelines and information for contributors.

## 🚀 **Getting Started**

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/festeraeb/CESARops.git
cd CESARops

# Install dependencies
pip install -r requirements.txt

# Run tests to ensure everything works
python test_ml_enhancements.py
```

### **Development Workflow**
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `python test_ml_enhancements.py`
4. Commit your changes: `git commit -m "Add your feature"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Create a Pull Request

## 📝 **Code Style**

### **Python Style**
- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

### **Documentation**
- Update README.md for new features
- Add docstrings to new functions
- Update requirements.txt for new dependencies
- Include examples in docstrings

## 🧪 **Testing**

### **Running Tests**
```bash
# Run the full test suite
python test_ml_enhancements.py

# Test specific components
python -c "from sarops import fetch_buoy_specifications; fetch_buoy_specifications()"
```

### **Adding Tests**
- Add new test functions to `test_ml_enhancements.py`
- Test both success and failure cases
- Include realistic test data
- Ensure tests are independent

## 🔧 **Adding New Features**

### **Data Sources**
When adding new environmental data sources:
1. Create a fetch function in `sarops.py`
2. Add database table schema
3. Update `auto_update_all_data()`
4. Add to the GUI if applicable
5. Update documentation

### **ML Enhancements**
For machine learning improvements:
1. Add training data collection
2. Implement model training functions
3. Update the `EnhancedOceanDrift` class
4. Add validation metrics
5. Document the approach

## 📊 **Data Management**

### **Database Schema**
- Use descriptive table and column names
- Include appropriate indexes
- Add UNIQUE constraints where needed
- Document schema changes

### **File Organization**
- Keep data files in `data/` directory
- Store models in `models/` directory
- Save outputs to `outputs/` directory
- Log to `logs/` directory

## 🐛 **Reporting Issues**

### **Bug Reports**
Please include:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Error messages and stack traces
- Log files from `logs/` directory

### **Feature Requests**
Please include:
- Use case description
- Expected benefits
- Implementation suggestions
- Related research or references

## 📚 **Resources**

### **Key Technologies**
- [OpenDrift Documentation](https://opendrift.github.io/)
- [NOAA Data Sources](https://www.noaa.gov/)
- [Scikit-learn ML](https://scikit-learn.org/)

### **SAR References**
- US Coast Guard SAR manuals
- NOAA drift modeling research
- Academic papers on oceanographic buoy drift

## 🤝 **Code of Conduct**

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on the merit of ideas, not individuals

## 📞 **Getting Help**

- Open an issue for questions
- Check existing issues and documentation first
- Provide context and be specific

Thank you for contributing to CESAROPS! Your efforts help improve search and rescue operations on the Great Lakes.