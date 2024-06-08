import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import src.Models.ontology_models as cura_models

# from your_module_path import CuraModelsBase  # Adjust the import according to your project structure

class TestCuraModelsBase(unittest.TestCase):
    def setUp(self):
        # Setup that runs before each test method
        self.cura_map_example = {'example': ['mapped_value']}
        self.cura_model = cura_models.CuraModelsBase(method='bert-base', cura_map=self.cura_map_example, from_tokenizer=True)
    
    def test_init(self):
        # Test initialization and attribute setting
        self.assertEqual(self.cura_model.method, 'bert-base')
        self.assertTrue(self.cura_model.from_tokenizer)
        self.assertIsNotNone(self.cura_model.cura_map)
        self.assertRaises(ValueError, cura_models.CuraModelsBase, 'unknown-method', self.cura_map_example, True)

    # Further tests can be added here
    @patch('src.Models.curation_models.CuraModelsBase.tokenizer', autospec=True)
    def test_get_tokenized(self, mock_tokenizer):
        # Mock the tokenizer to ensure no external calls are made
        query_list = ["test sentence"]
        self.cura_model.get_tokenized(query_list)
        mock_tokenizer.assert_called_once_with(query_list, padding=True, truncation=True, return_tensors='pt')

    @patch('src.Models.curation_models.CuraModelsBase.model', autospec=True)
    @patch('torch.no_grad', autospec=True)
    def test_create_embeddings_lm(self, mock_no_grad, mock_model):
        # Mock model to test embeddings creation without actual model inference
        query_list = ["test sentence"]
        embeddings = self.cura_model.create_embeddings_lm(query_list)
        mock_model.assert_called_once()  # Ensure the model was called
        self.assertIsInstance(embeddings, np.ndarray)  # Check the type of returned embeddings
        
if __name__ == '__main__':
    unittest.main()