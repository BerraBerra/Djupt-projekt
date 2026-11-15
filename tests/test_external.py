import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from prepare_external import difference_hash, pixel_hash
from evaluate_external import subgroup_results
from external_data_tools import RangeFile


class ExternalTests(unittest.TestCase):
    def test_range_reader_rejects_full_response_without_reading_body(self):
        response = MagicMock(status_code=200)
        with patch('external_data_tools.requests.Session') as factory:
            factory.return_value.get.return_value = response
            with self.assertRaisesRegex(ValueError, 'range request'):
                RangeFile('https://example.test/file')
            response.close.assert_called_once()
            self.assertTrue(factory.return_value.get.call_args.kwargs['stream'])

    def test_range_reader_seek_and_cached_reads(self):
        initial = MagicMock(status_code=206, url='https://example.test/file',
                            headers={'Content-Range': 'bytes 0-0/10'})
        content = MagicMock(status_code=206, content=b'345',
                            headers={'Content-Range': 'bytes 3-5/10'})
        with patch('external_data_tools.requests.Session') as factory:
            factory.return_value.get.side_effect = [initial, content]
            with RangeFile('https://example.test/file') as remote:
                remote.seek(3)
                self.assertEqual(remote.read(3), b'345')
                remote.seek(-7, 2)
                self.assertEqual(remote.read(3), b'345')
            self.assertEqual(factory.return_value.get.call_count, 2)

    def test_hashes_and_subgroups(self):
        image = Image.new('RGB', (224, 224), (50, 70, 90))
        self.assertEqual(pixel_hash(image), pixel_hash(image.copy()))
        self.assertEqual(difference_hash(image), difference_hash(image.resize((448,448))))
        metadata = [{'path':'a','label':'photo','real_source':'coco','generator':''},
                    {'path':'b','label':'ai_generated','generator':'g','real_source':'coco'},
                    {'path':'c','label':'ai_generated','generator':'g','real_source':'coco'}]
        predictions = [{'path':'a','predicted_label':'photo'},
                       {'path':'b','predicted_label':'photo'},
                       {'path':'c','predicted_label':'ai_generated'}]
        groups = subgroup_results(metadata, predictions)
        self.assertEqual(groups[0]['images'], 2)
        self.assertEqual(groups[0]['recall'], .5)
        self.assertEqual(groups[1]['recall'], 1)
