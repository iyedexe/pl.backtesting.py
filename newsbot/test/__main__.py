import unittest
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('error')
    suite = unittest.TestSuite()
    for mod in ('newsbot.test._test', 'newsbot.test._test_sources'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(mod))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
