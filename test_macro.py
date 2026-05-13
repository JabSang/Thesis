import folium
from branca.element import Template, MacroElement

m = folium.Map()
t = '''
{% macro html(this, kwargs) %}
<div id="thesis-hud">TEST</div>
{% endmacro %}
'''
macro = MacroElement()
macro._template = Template(t)
m.get_root().add_child(macro)
m.get_root().render()
html = m.get_root()._repr_html_()

with open('output_test.html', 'w') as f:
    f.write(html)

print("thesis-hud" in html)
