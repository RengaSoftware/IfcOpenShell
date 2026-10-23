pyinstaller -F ifcdiff.py ^
--add-data "..\ifcopenshell-python\ifcopenshell\util\*.json;ifcopenshell/util/" ^
--hidden-import="ifcopenshell.express.rules.IFC4" ^
--hidden-import="ifcopenshell.express" ^
--hidden-import="ifcopenshell.express.express_parser" ^
--hidden-import="ifcopenshell.express.express_parser_generated" ^
--collect-data="ifcopenshell" ^
--distpath .
