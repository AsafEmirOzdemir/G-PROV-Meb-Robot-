/**
 * @format
 */

import {AppRegistry} from 'react-native';
//import App from './App';
import AppPro from './AppPro';
import Main from './MainPage';
import {name as appName} from './app.json';
import App from './custombar';
AppRegistry.registerComponent(appName, () => Main);
